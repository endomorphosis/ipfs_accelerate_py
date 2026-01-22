#!/usr/bin/env python3
"""
Class-based test file for decoder-only models compatible with the refactored test suite.

This template provides a unified testing interface for decoder-only models like GPT
within the refactored test suite architecture, inheriting from ModelTest.
"""

import os
import sys
import json
import time
import datetime
import logging
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, Mock

# Import from the refactored test suite
from refactored_test_suite.model_test import ModelTest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models registry - Maps model IDs to their specific configurations
DECODER_MODELS_REGISTRY = {
    "gpt2": {
        "full_name": "GPT-2 (Small)",
        "architecture": "decoder-only",
        "description": "OpenAI GPT-2 Small model",
        "model_type": "gpt2",
        "parameters": "124M",
        "context_length": 1024,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "GPT2LMHeadModel",
        "tokenizer_class": "GPT2Tokenizer",
        "use_fast_tokenizer": True,
        "recommended_tasks": ["text-generation", "conversational"]
    },
    "gpt2-medium": {
        "full_name": "GPT-2 Medium",
        "architecture": "decoder-only",
        "description": "OpenAI GPT-2 Medium model",
        "model_type": "gpt2",
        "parameters": "355M",
        "context_length": 1024,
        "embedding_dim": 1024,
        "attention_heads": 16,
        "layers": 24,
        "model_class": "GPT2LMHeadModel",
        "tokenizer_class": "GPT2Tokenizer",
        "use_fast_tokenizer": True,
        "recommended_tasks": ["text-generation", "conversational"]
    },
    "facebook/opt-125m": {
        "full_name": "OPT-125M",
        "architecture": "decoder-only",
        "description": "Meta OPT model with 125M parameters",
        "model_type": "opt",
        "parameters": "125M",
        "context_length": 2048,
        "embedding_dim": 768,
        "attention_heads": 12,
        "layers": 12,
        "model_class": "OPTForCausalLM",
        "tokenizer_class": "GPT2Tokenizer",
        "use_fast_tokenizer": True,
        "recommended_tasks": ["text-generation", "conversational"]
    }
}

class TestDecoderModel(ModelTest):
    """Test class for decoder-only models like GPT-2, OPT, LLaMA, etc."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        
        # Initialize model-specific attributes
        self.model_id = "gpt2"
        
        # Verify model exists in registry
        if self.model_id not in DECODER_MODELS_REGISTRY:
            logger.warning(f"Model {self.model_id} not in registry, using default configuration")
            self.model_info = DECODER_MODELS_REGISTRY["gpt2"]
        else:
            self.model_info = DECODER_MODELS_REGISTRY[self.model_id]
        
        # Define model parameters
        self.task = "text-generation"
        self.model_class = self.model_info["model_class"]
        self.tokenizer_class = self.model_info["tokenizer_class"]
        self.description = self.model_info["description"]
        self.use_fast_tokenizer = self.model_info.get("use_fast_tokenizer", True)
        
        # Define test inputs
        self.test_text = "Once upon a time in a galaxy far, far away,"
        self.max_new_tokens = 20
        
        # Setup hardware detection
        self.setup_hardware()
    
    def setup_hardware(self):
        """Set up hardware detection."""
        try:
            # Try to import hardware detection capabilities
            from generators.hardware.hardware_detection import (
                HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
                detect_all_hardware
            )
            hardware_info = detect_all_hardware()
        except ImportError:
            # Fallback to manual detection
            import torch
            
            # Basic hardware detection
            self.has_cuda = torch.cuda.is_available()
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            self.has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            
            # Check for OpenVINO
            try:
                import openvino
                self.has_openvino = True
            except ImportError:
                self.has_openvino = False
            
            # WebNN/WebGPU are not directly accessible in Python
            self.has_webnn = False
            self.has_webgpu = False
        
        # Configure preferred device
        if self.has_cuda:
            self.device = 'cuda'
        elif self.has_mps:
            self.device = 'mps'
        elif self.has_rocm:
            self.device = 'cuda'  # ROCm uses CUDA compatibility layer
        else:
            self.device = 'cpu'
        
        logger.info(f"Using device: {self.device}")
    
    def tearDown(self):
        """Clean up resources after the test."""
        # Release any resources that need cleanup
        super().tearDown()
    
    def load_model(self, model_id=None):
        """Load the model for testing."""
        model_id = model_id or self.model_id
        
        try:
            import torch
            import transformers
            
            # Get model and tokenizer classes
            model_class = getattr(transformers, self.model_class)
            tokenizer_class = getattr(transformers, self.tokenizer_class, transformers.AutoTokenizer)
            
            # Load the tokenizer
            tokenizer = tokenizer_class.from_pretrained(
                model_id, 
                use_fast=self.use_fast_tokenizer
            )
            
            # Ensure the tokenizer has a pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load the model
            model = model_class.from_pretrained(model_id)
            
            # Move to appropriate device
            model = model.to(self.device)
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_input(self):
        """Prepare input for the model."""
        return self.test_text
    
    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_components = self.load_model()
        
        # Verify that model and tokenizer were loaded
        self.assertIsNotNone(model_components["model"])
        self.assertIsNotNone(model_components["tokenizer"])
        
        logger.info("Model loaded successfully")
    
    def test_basic_inference(self):
        """Test basic inference with the model."""
        import torch
        
        # Load model
        model_components = self.load_model()
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        
        # Prepare input
        input_text = self.prepare_input()
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to device if needed
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Verify outputs
        self.assertIsNotNone(outputs)
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check that the output contains the input text and has been extended
        self.assertTrue(input_text in generated_text)
        self.assertGreater(len(generated_text), len(input_text))
        
        logger.info(f"Generated text: {generated_text}")
        logger.info("Basic inference successful")
    
    def test_pipeline_inference(self):
        """Test inference using the pipeline API."""
        try:
            import transformers
            
            # Initialize the pipeline
            pipe = transformers.pipeline(
                self.task, 
                model=self.model_id,
                device=self.device if self.device != "cpu" else -1
            )
            
            # Test with input text
            input_text = self.prepare_input()
            
            # Run inference
            outputs = pipe(input_text, max_new_tokens=self.max_new_tokens, do_sample=True)
            
            # Verify outputs
            self.assertIsNotNone(outputs)
            
            # Log results
            if self.task == "text-generation":
                generated_text = outputs[0]["generated_text"] if isinstance(outputs, list) else outputs["generated_text"]
                logger.info(f"Generated text: {generated_text}")
            else:
                logger.info(f"Pipeline output: {outputs[:2]}")
            
            logger.info("Pipeline inference successful")
            
        except Exception as e:
            logger.error(f"Error in pipeline inference: {e}")
            self.fail(f"Pipeline inference failed: {e}")
    
    def test_hardware_compatibility(self):
        """Test the model's compatibility with different hardware platforms."""
        devices_to_test = []
        
        # Add available devices
        if self.has_cuda:
            devices_to_test.append('cuda')
        if self.has_mps:
            devices_to_test.append('mps')
        
        # Always test CPU
        if 'cpu' not in devices_to_test:
            devices_to_test.append('cpu')
        
        results = {}
        
        # Test on each device
        for device in devices_to_test:
            original_device = self.device
            try:
                logger.info(f"Testing on {device}...")
                self.device = device
                
                # Load model and prepare input
                model_components = self.load_model()
                model = model_components["model"]
                tokenizer = model_components["tokenizer"]
                
                input_text = self.prepare_input()
                inputs = tokenizer(input_text, return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run inference with minimal tokens to save time
                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,  # Use fewer tokens for hardware compatibility test
                        num_return_sequences=1
                    )
                
                # Verify results
                results[device] = True
                logger.info(f"Test on {device} successful")
                
            except Exception as e:
                logger.error(f"Error testing on {device}: {e}")
                results[device] = False
            finally:
                # Restore original device
                self.device = original_device
        
        # Verify at least one device works
        self.assertTrue(any(results.values()), "Model should work on at least one device")
        
        # Log results
        for device, success in results.items():
            logger.info(f"Device {device}: {'Success' if success else 'Failed'}")
    
    def test_streaming_generation(self):
        """Test streaming text generation."""
        try:
            import torch
            import transformers
            
            # Load model components
            model_components = self.load_model()
            model = model_components["model"]
            tokenizer = model_components["tokenizer"]
            
            # Prepare input
            input_text = self.prepare_input()
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Set up streaming with lower token count for test
            streamer = transformers.TextStreamer(tokenizer)
            
            # Log start of streaming
            logger.info("Starting streaming generation (tokens will be shown one by one):")
            
            # Stream tokens for small number to save time
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    streamer=streamer,
                    do_sample=True,
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify the output
            self.assertTrue(input_text in generated_text)
            self.assertGreater(len(generated_text), len(input_text))
            
            logger.info(f"\nFull generated text: {generated_text}")
            logger.info("Streaming generation successful")
            
        except ImportError as e:
            logger.warning(f"Streaming test skipped - transformers may need to be updated: {e}")
            self.skipTest(f"Streaming functionality not available: {e}")
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            self.fail(f"Streaming generation failed: {e}")
    
    def run_all_tests(self):
        """Run all tests for this model."""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        results = {}
        
        for method in test_methods:
            try:
                logger.info(f"Running {method}...")
                getattr(self, method)()
                results[method] = "PASS"
            except Exception as e:
                logger.error(f"Error in {method}: {e}")
                results[method] = f"FAIL: {str(e)}"
        
        return results


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test decoder-only models with refactored test suite")
    parser.add_argument("--model", type=str, default="gpt2", 
                       help="Model ID to test")
    parser.add_argument("--device", type=str, help="Device to test on (cpu, cuda, etc.)")
    parser.add_argument("--task", type=str, default="text-generation", 
                       help="Task to test (text-generation, conversational, etc.)")
    parser.add_argument("--max-tokens", type=int, default=20, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Create test instance
    test = TestDecoderModel()
    
    # Override settings if specified
    if args.model:
        test.model_id = args.model
    if args.device:
        test.device = args.device
    if args.task:
        test.task = args.task
    if args.max_tokens:
        test.max_new_tokens = args.max_tokens
    
    # Run tests
    test.setUp()
    results = test.run_all_tests()
    test.tearDown()
    
    # Print results
    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # Save results if requested
    if args.save_results:
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{args.model.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump({
                "model": args.model,
                "device": test.device,
                "task": test.task,
                "max_tokens": test.max_new_tokens,
                "results": results,
                "timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# Base template for Hugging Face model tests
# This serves as the root template for all other templates

"""TEMPLATE_METADATA
name: hf_template
type: base
description: Base template for all Hugging Face model tests
provides: base_structure, resource_pool_integration, hardware_detection
requires: os, sys, torch, transformers, resource_pool
compatible_with: cpu, cuda, rocm, mps, openvino
author: IPFS Accelerate Python Framework Team
"""

import os
import sys
import torch
import logging
import argparse
from transformers import AutoModel, AutoTokenizer

# Ensure resource_pool is available by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SECTION: class_definition
class TestHuggingFaceModel:
    """Base test class for Hugging Face models"""
    
    # Model name to load - this should be overridden by derived templates
    model_name = "{{ model_name }}"
    
    @classmethod
    def setup_class(cls):
        """Set up test class - load model once for all tests"""
        # Use resource pool to efficiently share resources
        pool = get_global_resource_pool()
        
        # Get dependencies with resource pooling
        torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Create model constructor
        def create_model():
            model = AutoModel.from_pretrained(cls.model_name)
            logger.info(f"Model {cls.model_name} loaded successfully")
            return model
        
        # Get or create model
        cls.model = pool.get_model("huggingface", cls.model_name, constructor=create_model)
        
        # Create tokenizer constructor
        def create_tokenizer():
            tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
            return tokenizer
        
        # Get or create tokenizer
        cls.tokenizer = pool.get_tokenizer("huggingface", cls.model_name, constructor=create_tokenizer)
        
        # Set device based on available hardware
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            cls.device = torch.device("cuda")
        else:
            logger.info("CUDA not available, using CPU")
            cls.device = torch.device("cpu")
        
        # Move model to device
        cls.model = cls.model.to(cls.device)
        
        logger.info(f"Setup complete for {cls.model_name} on {cls.device}")
    
    # SECTION: test_model_loading
    def test_model_loading(self):
        """Test that the model was loaded correctly"""
        assert self.model is not None, "Model should be loaded"
        assert self.tokenizer is not None, "Tokenizer should be loaded"
        
        # Check model type - implemented by subclasses
        self._check_model_type()
    
    # SECTION: check_model_type
    def _check_model_type(self):
        """Check model type - to be implemented by subclasses"""
        # Generic check
        assert hasattr(self.model, "forward"), "Model should have a forward method"
    
    # SECTION: test_basic_inference
    def test_basic_inference(self):
        """Test basic model inference"""
        # Prepare input
        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Move input to proper device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Check outputs
        assert outputs is not None, "Outputs should not be None"
        
        # Additional checks implemented by subclasses
        self._check_outputs(outputs)
    
    # SECTION: check_outputs
    def _check_outputs(self, outputs):
        """Check model outputs - to be implemented by subclasses"""
        # Generic check
        assert len(outputs) > 0, "Model should produce at least one output"
    
    # SECTION: test_device_handling
    def test_device_handling(self):
        """Test device handling (CPU/CUDA/etc.)"""
        # Check that model is on correct device
        if hasattr(self.model, "device"):
            assert self.model.device == self.device, f"Model should be on {self.device}"
        else:
            # Check first parameter
            param = next(self.model.parameters())
            assert param.device == self.device, f"Model parameters should be on {self.device}"
        
        # Test with CPU
        model_cpu = self.model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu", "Model should be on CPU"
        
        # Move back to original device
        self.model = self.model.to(self.device)
    
    # SECTION: test_cuda_support
    def test_cuda_support(self):
        """Test CUDA support if available"""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping test")
            return
        
        # Model should be able to move to CUDA
        model_cuda = self.model.to("cuda")
        
        # Prepare input
        text = "Testing CUDA support"
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model_cuda(**inputs)
        
        # Check outputs are on CUDA
        assert next(iter(outputs.values())).device.type == "cuda", "Outputs should be on CUDA"
    
    # SECTION: teardown_class
    @classmethod
    def teardown_class(cls):
        """Clean up resources"""
        # Report stats from resource pool
        pool = get_global_resource_pool()
        stats = pool.get_stats()
        logger.info(f"Resource pool stats: {stats}")
        
        # Explicitly clear model reference
        cls.model = None
        cls.tokenizer = None
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Try to clear CUDA cache if available
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        
        logger.info("Test teardown complete")

# SECTION: main_function
def main():
    """Main function to run tests directly"""
    # Parse arguments
    parser = argparse.ArgumentParser(description=f"Test for {{ model_name }}")
    parser.add_argument("--model", type=str, default="{{ model_name }}", 
                       help="Model name to test")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                       help="Device to run tests on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Override model name if provided
    TestHuggingFaceModel.model_name = args.model
    
    # Set up test
    test = TestHuggingFaceModel()
    TestHuggingFaceModel.setup_class()
    
    # Run tests
    try:
        logger.info("Running model loading test")
        test.test_model_loading()
        
        logger.info("Running basic inference test")
        test.test_basic_inference()
        
        logger.info("Running device handling test")
        test.test_device_handling()
        
        if args.device != "cpu" and torch.cuda.is_available():
            logger.info("Running CUDA support test")
            test.test_cuda_support()
        
        logger.info("All tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up
        TestHuggingFaceModel.teardown_class()

if __name__ == "__main__":
    main()
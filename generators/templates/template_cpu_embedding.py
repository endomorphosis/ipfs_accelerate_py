#!/usr/bin/env python
'''
CPU-optimized hardware test template for embedding models like BERT
'''

import os
import sys
import unittest
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import resource pool
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from resource_pool import get_global_resource_pool

class Test{model_class}OnCPU(unittest.TestCase):
    '''CPU-optimized test for {model_name} embedding model'''
    
    @classmethod
    def setUpClass(cls):
        '''Set up the test class - load model once for all tests'''
        # Get resource pool
        pool = get_global_resource_pool()
        
        # Load required libraries
        cls.torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        cls.transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Define model constructor with CPU optimizations
        def create_model():
            from transformers import {auto_class}
            
            # Set appropriate model config for improved CPU performance
            config_kwargs = {{
                "torchscript": True,  # Enable TorchScript for better CPU performance
                "return_dict": False  # Slightly faster without dictionary outputs
            }}
            
            model = {auto_class}.from_pretrained(
                "{model_name}", 
                **config_kwargs
            )
            
            # Optimize for CPU inference if possible
            try:
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model)
                )
                logger.info("Successfully applied TorchScript optimization")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {{e}}")
                # Still use the original model
            
            return model
        
        # Set hardware preferences
        hardware_preferences = {{
            "device": "cpu",
            "precision": "fp32"  # Use full precision for CPU
        }}
        
        # Load model with hardware preferences
        cls.model = pool.get_model(
            "{model_family}",
            "{model_name}",
            constructor=create_model,
            hardware_preferences=hardware_preferences
        )
        
        # Define tokenizer constructor
        def create_tokenizer():
            from transformers import {tokenizer_class}
            return {tokenizer_class}.from_pretrained("{model_name}")
        
        # Load tokenizer
        cls.tokenizer = pool.get_tokenizer(
            "{model_family}",
            "{model_name}",
            constructor=create_tokenizer
        )
        
        # Verify resources loaded correctly
        assert cls.model is not None, "Failed to load model"
        assert cls.tokenizer is not None, "Failed to load tokenizer"
        
        # Get model device
        if hasattr(cls.model, "device"):
            cls.device = cls.model.device
        else:
            # Try to get device from model parameters
            try:
                cls.device = next(cls.model.parameters()).device
            except:
                # Fallback to CPU
                cls.device = cls.torch.device("cpu")
        
        # Log model and device information
        logger.info(f"Model loaded: {{type(cls.model).__name__}}")
        logger.info(f"Device: {{cls.device}}")
        
        # Enable multicore CPU acceleration if available
        try:
            import torch.backends.mkldnn
            if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.enabled:
                logger.info("MKL-DNN acceleration enabled")
            else:
                logger.info("MKL-DNN acceleration not available")
                
            # Set number of threads for optimal performance
            num_threads = cls.torch.get_num_threads()
            logger.info(f"PyTorch using {{num_threads}} CPU threads")
            
            # Set thread locality for better CPU cache usage
            if hasattr(cls.torch, 'set_num_interop_threads'):
                # For parallel model execution (default: 2-4)
                cls.torch.set_num_interop_threads(max(2, os.cpu_count() // 4))
                # For within operations (default: max cores)
                cls.torch.set_num_threads(os.cpu_count())
                logger.info(f"Thread configuration adjusted for CPU optimization")
        except Exception as e:
            logger.warning(f"Error configuring CPU acceleration: {{e}}")
    
    def test_model_on_cpu(self):
        '''Test that the model is on CPU'''
        device_type = str(self.device).split(':')[0]
        self.assertEqual(device_type, "cpu", 
                       f"Model should be on CPU, but is on {{device_type}}")
    
    def test_basic_inference(self):
        '''Test basic inference functionality'''
        # Create a simple input
        text = "This is a test"
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Make sure inputs are on CPU
        inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
        
        # Run inference with performance measurement
        start_time = self.torch.backends.cudnn.benchmark
        with self.torch.no_grad():
            # Time the forward pass
            start = time.time()
            outputs = self.model(**inputs)
            end = time.time()
        
        # Get the output tensor - works with both return_dict=True and False
        if isinstance(outputs, tuple):
            # Model was configured with return_dict=False
            hidden_states = outputs[0]
        else:
            # Model returns a dictionary-like object
            hidden_states = outputs.last_hidden_state
        
        # Verify output shape
        self.assertIsNotNone(hidden_states, "Model output should not be None")
        self.assertEqual(hidden_states.shape[0], 1, "Batch size should be 1")
        
        # Log performance
        inference_time = (end - start) * 1000  # Convert to ms
        logger.info(f"Inference time: {{inference_time:.2f}} ms")
    
    def test_batch_processing(self):
        '''Test batch processing on CPU'''
        # Create a small batch of inputs
        texts = ["This is the first example", 
                "This is the second example",
                "And this is the third one"]
                
        # Tokenize with padding
        batch_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        
        # Make sure batch is on CPU
        batch_inputs = {{k: v.to(self.device) for k, v in batch_inputs.items()}}
        
        # Process batch
        with self.torch.no_grad():
            batch_outputs = self.model(**batch_inputs)
        
        # Verify output dimensions
        if isinstance(batch_outputs, tuple):
            # Model was configured with return_dict=False
            batch_hidden_states = batch_outputs[0]
        else:
            # Model returns a dictionary-like object
            batch_hidden_states = batch_outputs.last_hidden_state
            
        self.assertEqual(batch_hidden_states.shape[0], 3, 
                       "Output batch size should match input batch size")
    
    def test_memory_efficiency(self):
        '''Test memory efficiency on CPU'''
        # Check initial memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run an inference pass
        texts = ["This is a test"] * 10  # Small batch of identical texts
        batch_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        batch_inputs = {{k: v.to(self.device) for k, v in batch_inputs.items()}}
        
        # Force garbage collection before inference
        import gc
        gc.collect()
        
        # Run inference
        with self.torch.no_grad():
            _ = self.model(**batch_inputs)
        
        # Check memory after inference
        gc.collect()  # Force garbage collection
        post_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Log memory usage
        logger.info(f"Memory before: {{initial_memory:.2f}} MB")
        logger.info(f"Memory after: {{post_memory:.2f}} MB")
        logger.info(f"Memory increase: {{post_memory - initial_memory:.2f}} MB")
        
        # No hard assertion since memory usage varies, just log it
    
    @classmethod
    def tearDownClass(cls):
        '''Clean up resources'''
        # Get resource pool stats
        pool = get_global_resource_pool()
        stats = pool.get_stats()
        logger.info(f"Resource pool stats: {{stats}}")
        
        # Cleanup unused resources
        pool.cleanup_unused_resources(max_age_minutes=0.1)  # 6 seconds

def main():
    '''Run the tests'''
    unittest.main()

if __name__ == "__main__":
    import time  # For performance measurements
    main()
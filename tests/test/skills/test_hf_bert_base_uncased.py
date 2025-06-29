#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import hardware detection capabilities if available:
try:
    from generators.hardware.hardware_detection import ()
    HAS_CUDA, HAS_ROCM, HAS_OPENVINO, HAS_MPS, HAS_WEBNN, HAS_WEBGPU,
    detect_all_hardware
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    HAS_HARDWARE_DETECTION = False
    # We'll detect hardware manually as fallback
    """
    Test file for bert-base-uncased
    Generated automatically by test_generator_with_resource_pool.py
    Model family: embedding
    Generated on: 2025-03-02T18:57:24.395490
    """

    import os
    import sys
    import logging
    import unittest
    from typing import Dict, List, Any, Optional

# Import the resource pool for efficient resource sharing
    sys.path.append()os.path.dirname()os.path.dirname()os.path.abspath()__file__))))
    from resource_pool import get_global_resource_pool

# Set up logging
    logging.basicConfig()level=logging.INFO,
    format='%()asctime)s - %()name)s - %()levelname)s - %()message)s')
    logger = logging.getLogger()__name__)

class TestHFBert()unittest.TestCase):
    """Test class for bert-base-uncased ()embedding model)"""
    
    @classmethod
    def setUpClass()cls):
        """Set up the test class - load model once for all tests"""
        # Use resource pool to efficiently share resources
        pool = get_global_resource_pool())
        
        # Load dependencies
        cls.torch = pool.get_resource()"torch", constructor=lambda: __import__()"torch"))
        cls.transformers = pool.get_resource()"transformers", constructor=lambda: __import__()"transformers"))
        
        # Define model constructor
        def create_model()):
            from transformers import AutoModel
        return AutoModel.from_pretrained()"bert-base-uncased")
        
        # Set hardware preferences for optimal hardware selection
        hardware_preferences = {}
        "device": "cpu"
        }
        
        # Get or create model from pool with hardware awareness
        cls.model = pool.get_model()"embedding", "bert-base-uncased", 
        constructor=create_model,
        hardware_preferences=hardware_preferences)
        
        # Define tokenizer constructor
        def create_tokenizer()):
            from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained()"bert-base-uncased")
        
        # Get or create tokenizer from pool
        cls.tokenizer = pool.get_tokenizer()"embedding", "bert-base-uncased", 
        constructor=create_tokenizer)
        
        # Verify resources loaded correctly
        assert cls.model is not None, "Failed to load model"
        assert cls.tokenizer is not None, "Failed to load tokenizer/processor"
        
        # Get device - use model device if available:, otherwise use preferred device
        if hasattr()cls.model, "device"):
            cls.device = cls.model.device
        else:
            cls.device = cls.torch.device()"cpu")
            # Move model to device if needed:::
            if hasattr()cls.model, "to"):
                cls.model = cls.model.to()cls.device)
        
                logger.info()f"Model loaded on device: {}cls.device}")
    
    def test_model_loading()self):
        """Test that the model was loaded correctly"""
        self.assertIsNotNone()self.model, "Model should be loaded")
        self.assertIsNotNone()self.tokenizer, "Tokenizer/processor should be loaded")
        
        # Check model type
        from transformers import AutoModel
        expected_class = AutoModel.from_pretrained()"bert-base-uncased").__class__
        self.assertIsInstance()self.model, expected_class, 
        f"Model should be an instance of {}expected_class.__name__}")
    
    
    def test_embedding_generation()self):
        """Test embedding generation"""
        # Prepare input
        text = "Hello, world!"
        inputs = self.tokenizer()text, return_tensors="pt")
        
        # Move inputs to device if needed:::
        inputs = {}k: v.to()self.device) for k, v in inputs.items())}
        
        # Run inference
        with self.torch.no_grad()):
            outputs = self.model()**inputs)
        
        # Check outputs
            self.assertIsNotNone()outputs, "Outputs should not be None")
            self.assertTrue()hasattr()outputs, "last_hidden_state"),
            "Output should have last_hidden_state attribute")
        
        # Check embedding shape
            last_hidden_state = outputs.last_hidden_state
            self.assertEqual()last_hidden_state.shape[0], 1, "Batch size should be 1")
        
            ,
    def test_output_shapes()self):
        """Test that model outputs have expected shapes"""
        # Prepare a basic input
        
        text = "Hello, world!"
        inputs = self.tokenizer()text, return_tensors="pt")
        
        # Move inputs to device if needed:::
        inputs = {}k: v.to()self.device) for k, v in inputs.items())}
        
        # Run inference
        with self.torch.no_grad()):
            outputs = self.model()**inputs)
        
        # Check output shapes
        
            self.assertTrue()hasattr()outputs, "last_hidden_state"), "Output should have last_hidden_state attribute")
            self.assertEqual()outputs.last_hidden_state.shape, ()1, 6, 768),
            f"Output last_hidden_state shape should be [1, 6, 768], got {}outputs.last_hidden_state.shape}")
            ,
            self.assertTrue()hasattr()outputs, "pooler_output"), "Output should have pooler_output attribute")
            self.assertEqual()outputs.pooler_output.shape, ()1, 768),
            f"Output pooler_output shape should be [1, 768], got {}outputs.pooler_output.shape}")
        
            ,
    def test_device_compatibility()self):
        """Test device compatibility"""
        device_str = str()self.device)
        logger.info()f"Testing on device: {}device_str}")
        
        # Check model device
        if hasattr()self.model, "device"):
            model_device = str()self.model.device)
            logger.info()f"Model is on device: {}model_device}")
            self.assertEqual()model_device, device_str, 
            f"Model should be on {}device_str}, but is on {}model_device}")
        
        # Check parameter device
        if hasattr()self.model, "parameters"):
            try:
                param_device = str()next()self.model.parameters())).device)
                logger.info()f"Model parameters are on device: {}param_device}")
                self.assertEqual()param_device, device_str, 
                f"Parameters should be on {}device_str}, but are on {}param_device}")
            except Exception as e:
                logger.warning()f"Could not check parameter device: {}e}")
    
                @classmethod
    def tearDownClass()cls):
        """Clean up resources"""
        # Get resource pool stats
        pool = get_global_resource_pool())
        stats = pool.get_stats())
        logger.info()f"Resource pool stats: {}stats}")
        
        # Clean up unused resources to prevent memory leaks
        pool.cleanup_unused_resources()max_age_minutes=0.1)  # 6 seconds

def main()):
    """Run the test directly"""
    unittest.main())

if __name__ == "__main__":
    main())
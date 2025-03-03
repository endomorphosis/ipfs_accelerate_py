#!/usr/bin/env python
# Automatically generated test file for bert-base-uncased
# Generated on: 2025-03-02T16:58:04.232748

import os
import sys
import torch
import logging
import argparse
from transformers import AutoModel, AutoTokenizer

# Import the resource pool for efficient resource sharing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_pool import get_global_resource_pool

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestHFBertModel:
    """Test class for bert-base-uncased"""
    
    @classmethod
    def setup_class(cls):
        """Set up the test class - load model once for all tests"""
        # Use resource pool to efficiently share resources
        pool = get_global_resource_pool()
        
        # Load dependencies
        torch = pool.get_resource("torch", constructor=lambda: __import__("torch"))
        transformers = pool.get_resource("transformers", constructor=lambda: __import__("transformers"))
        
        # Define model constructor
        def create_model():
            return AutoModel.from_pretrained("bert-base-uncased")
        
        # Get or create model from pool
        cls.model = pool.get_model("BertModel", "bert-base-uncased", constructor=create_model)
        
        # Define tokenizer constructor
        def create_tokenizer():
            return AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Get or create tokenizer from pool
        cls.tokenizer = pool.get_tokenizer("BertModel", "bert-base-uncased", constructor=create_tokenizer)
        
        # Check if model and tokenizer were loaded successfully
        assert cls.model is not None, "Failed to load model"
        assert cls.tokenizer is not None, "Failed to load tokenizer"
        
        # Set device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = cls.model.to(cls.device)
        
        logger.info(f"Test setup complete for bert-base-uncased")
    
    def test_model_loading(self):
        """Test that the model was loaded correctly"""
        assert self.model is not None, "Model should be loaded"
        assert self.tokenizer is not None, "Tokenizer should be loaded"
        assert isinstance(self.model, AutoModel.from_pretrained("bert-base-uncased").__class__), "Model should be the correct type"
    
    def test_basic_inference(self):
        """Test basic inference"""
        # Prepare input
        text = "Hello, world!"
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Validate outputs
        assert outputs is not None, "Outputs should not be None"
        
        # Check output shapes
        assert hasattr(outputs, "last_hidden_state"), "Output should have last_hidden_state attribute"
        assert tuple(outputs.last_hidden_state.shape) == tuple([1, 6, 768]), f"Output last_hidden_state shape should be [1, 6, 768], got {outputs.last_hidden_state.shape}"
        
        # Check output shapes
        assert hasattr(outputs, "pooler_output"), "Output should have pooler_output attribute"
        assert tuple(outputs.pooler_output.shape) == tuple([1, 768]), f"Output pooler_output shape should be [1, 768], got {outputs.pooler_output.shape}"
        
    def test_cuda_support(self):
        """Test CUDA support if available"""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping test")
            return
        
        # Move model to CUDA
        self.model = self.model.to("cuda")
        
        # Prepare input
        text = "Testing CUDA support"
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Validate that outputs are on CUDA
        assert outputs is not None, "Outputs should not be None"
        assert next(iter(outputs.values())).device.type == "cuda", "Outputs should be on CUDA"
    
    @classmethod
    def teardown_class(cls):
        """Clean up resources - this is less important with ResourcePool"""
        # The ResourcePool manages resource cleanup, but we can clear specific
        # references to ensure better garbage collection
        cls.model = None
        cls.tokenizer = None
        logger.info("Test teardown complete")

def main():
    """Run the test directly"""
    # Parse arguments
    model_name = "bert-base-uncased"
    parser = argparse.ArgumentParser(description=f"Test for {model_name}")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                       help="Device to run tests on")
    args = parser.parse_args()
    
    # Set up test
    test = TestHFBertModel()
    TestHFBertModel.setup_class()
    
    # Run tests
    try:
        logger.info("Running model loading test")
        test.test_model_loading()
        
        logger.info("Running basic inference test")
        test.test_basic_inference()
        
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
        TestHFBertModel.teardown_class()
        
        # Get resource pool stats
        pool = get_global_resource_pool()
        stats = pool.get_stats()
        logger.info(f"Resource pool stats: {stats}")

if __name__ == "__main__":
    main()
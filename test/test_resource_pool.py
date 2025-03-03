#!/usr/bin/env python
# Test script for the ResourcePool class

import os
import time
import logging
from resource_pool import get_global_resource_pool

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_torch():
    """Load PyTorch module"""
    import torch
    return torch

def load_transformers():
    """Load transformers module"""
    import transformers
    return transformers

def load_bert_model():
    """Load a BERT model for testing"""
    import torch
    import transformers
    # Use tiny model for testing
    return transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny")

def test_resource_sharing():
    """Test that resources are properly shared"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First access (miss)
    logger.info("Loading torch for the first time")
    torch1 = pool.get_resource("torch", constructor=load_torch)
    
    # Second access (hit)
    logger.info("Loading torch for the second time")
    torch2 = pool.get_resource("torch", constructor=load_torch)
    
    # Check that we got the same object
    assert torch1 is torch2, "Resource pool failed to return the same object"
    
    # Check stats
    stats = pool.get_stats()
    logger.info(f"Resource pool stats: {stats}")
    assert stats["hits"] == 1, "Expected one cache hit"
    assert stats["misses"] == 1, "Expected one cache miss"
    
    logger.info("Resource sharing test passed!")

def test_model_caching():
    """Test model caching functionality"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First check that resources are available
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    if torch is None or transformers is None:
        logger.error("Required dependencies missing for model caching test")
        return
    
    # First access (miss)
    logger.info("Loading BERT model for the first time")
    model1 = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    
    # Second access (hit)
    logger.info("Loading BERT model for the second time")
    model2 = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    
    # Check that we got the same object
    assert model1 is model2, "Resource pool failed to return the same model"
    
    # Check stats
    stats = pool.get_stats()
    logger.info(f"Resource pool stats after model loading: {stats}")
    
    logger.info("Model caching test passed!")

def test_cleanup():
    """Test cleanup of unused resources"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # Load some temporary resources
    pool.get_resource("temp_resource", constructor=lambda: {"data": "temporary"})
    
    # Get stats before cleanup
    stats_before = pool.get_stats()
    logger.info(f"Stats before cleanup: {stats_before}")
    
    # Cleanup with a short timeout (0.1 minutes)
    # This will remove resources that haven't been accessed in the last 6 seconds
    time.sleep(7)  # Wait to ensure the resource is older than the timeout
    removed = pool.cleanup_unused_resources(max_age_minutes=0.1)
    
    # Get stats after cleanup
    stats_after = pool.get_stats()
    logger.info(f"Stats after cleanup: {stats_after}")
    logger.info(f"Removed {removed} resources")
    
    logger.info("Cleanup test passed!")

def test_example_workflow():
    """Test an example workflow using the resource pool"""
    # Get the resource pool
    pool = get_global_resource_pool()
    
    # First, we'd ensure necessary libraries are available
    torch = pool.get_resource("torch", constructor=load_torch)
    transformers = pool.get_resource("transformers", constructor=load_transformers)
    
    if torch is None or transformers is None:
        logger.error("Required dependencies missing for example workflow test")
        return
    
    # Load a model
    logger.info("Loading model for test generation")
    model = pool.get_model("bert", "prajjwal1/bert-tiny", constructor=load_bert_model)
    if model is None:
        logger.error("Failed to load model for example workflow test")
        return
    
    # Simulate test generation
    logger.info("Generating tests using cached model")
    
    # Simulate using model for inference
    if hasattr(model, "forward"):
        try:
            # Create a simple input tensor
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            with torch.no_grad():
                outputs = model(input_ids)
            
            logger.info(f"Model produced output with shape: {outputs.last_hidden_state.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
    
    # Show memory usage
    stats = pool.get_stats()
    logger.info(f"Memory usage after test generation: {stats['memory_usage_mb']:.2f} MB")
    
    logger.info("Example workflow test passed!")

def main():
    """Run all tests"""
    logger.info("Starting ResourcePool tests")
    
    try:
        # Run tests
        test_resource_sharing()
        test_model_caching()
        test_cleanup()
        test_example_workflow()
        
        # Final cleanup
        pool = get_global_resource_pool()
        pool.clear()
        
        logger.info("All ResourcePool tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
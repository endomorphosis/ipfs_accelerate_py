#!/usr/bin/env python3
"""
Template for testing vision models (ViT, ResNet, etc.) on different hardware platforms.

This template includes placeholders that will be replaced with actual values during test generation:
- ${model_name}: Name of the model (e.g., "vit-base-patch16-224")
- ${model_family}: Family of the model (e.g., "vision")
- ${hardware_type}: Type of hardware to run on (e.g., "cpu", "cuda", "rocm")
- ${batch_size}: Batch size for testing (e.g., 1, 4, 8)
"""

import os
import time
import logging
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_${model_family}")

def test_${model_name.replace("-", "_").replace("/", "_")}_on_${hardware_type}():
    """Test ${model_name} model on ${hardware_type} hardware with batch size ${batch_size}."""
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Log test configuration
    logger.info(f"Testing model: ${model_name}")
    logger.info(f"Hardware: ${hardware_type}")
    logger.info(f"Batch size: ${batch_size}")
    
    try:
        # Initialize processor and model
        processor = AutoImageProcessor.from_pretrained("${model_name}")
        model = AutoModel.from_pretrained("${model_name}")
        
        # Move model to appropriate device
        device = "${hardware_type}"
        if device != "cpu":
            model = model.to(device)
        
        # Log model information
        logger.info(f"Model loaded successfully: {model.__class__.__name__}")
        
        # Create dummy image for testing
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Process the image
        inputs = processor(dummy_image, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create batch by repeating the input
        # Handle different tensor shapes appropriately
        batch_inputs = {}
        for k, v in inputs.items():
            if v.dim() == 4:  # Image tensor (B, C, H, W)
                batch_inputs[k] = v.repeat(${batch_size}, 1, 1, 1)
            else:  # Other tensors
                batch_inputs[k] = v.repeat(${batch_size}, 1)
        
        # Measure inference time
        inference_start = time.time()
        
        # Run inference
        with torch.no_grad():
            outputs = model(**batch_inputs)
        
        # Calculate inference time
        inference_time = time.time() - inference_start
        
        # Extract features
        features = outputs.last_hidden_state
        
        # Log results
        logger.info(f"Inference completed in {inference_time:.4f} seconds")
        logger.info(f"Output shape: {features.shape}")
        
        # Measure memory usage
        if device == "cuda":
            memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            import psutil
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # MB
            
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Return test results
        return {
            "test_id": "${test_id}",  # Will be replaced with a UUID during generation
            "model_name": "${model_name}",
            "model_family": "${model_family}",
            "hardware_type": "${hardware_type}",
            "batch_size": ${batch_size},
            "execution_time": execution_time,
            "inference_time": inference_time,
            "memory_usage": memory_usage,
            "feature_shape": features.shape,
            "success": True
        }
        
    except Exception as e:
        # Log error and return failure result
        logger.error(f"Test failed: {str(e)}")
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        return {
            "test_id": "${test_id}",
            "model_name": "${model_name}",
            "model_family": "${model_family}",
            "hardware_type": "${hardware_type}",
            "batch_size": ${batch_size},
            "execution_time": execution_time,
            "success": False,
            "error_message": str(e)
        }

if __name__ == "__main__":
    # This allows the template to be run directly for testing
    result = test_${model_name.replace("-", "_").replace("/", "_")}_on_${hardware_type}()
    print(f"Test result: {'Success' if result['success'] else 'Failure'}")
    if result['success']:
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        print(f"Inference time: {result['inference_time']:.4f} seconds")
        print(f"Memory usage: {result['memory_usage']:.2f} MB")
        print(f"Feature shape: {result['feature_shape']}")